(function () {
  const PHOTO_INTERVAL_MS = 7000;
  const POLL_INTERVAL_MS = 15000;
  const logReceiverError = (payload) => {
    try {
      fetch("/cast/log", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    } catch (err) {
      console.warn("Failed to report cast error", err);
    }
  };

  window.onerror = function (message, source, lineno, colno, error) {
    logReceiverError({
      message,
      source,
      lineno,
      colno,
      stack: error && error.stack,
    });
  };

  window.addEventListener("unhandledrejection", (event) => {
    logReceiverError({
      message: event.reason && event.reason.message,
      stack: event.reason && event.reason.stack,
    });
  });

  const initialState = window.CAST_STATE || { event: {}, photos: [], games: [], generated_at: null };

  const layoutEl = document.querySelector(".cast-layout");
  const frameEl = document.querySelector(".cast-photo__frame");
  let photoEl = document.querySelector("[data-cast-photo]");
  let placeholderEl = document.querySelector("[data-cast-photo-placeholder]");
  const captionEl = document.querySelector("[data-photo-caption]");
  const gamesContainer = document.querySelector("[data-cast-games]");
  const updatedEl = document.querySelector("[data-last-updated]");
  const leaderboardSection = document.querySelector("[data-cast-leaderboard]");
  const leaderboardGrid = document.querySelector("[data-cast-leaderboard-grid]");

  let photos = Array.isArray(initialState.photos) ? initialState.photos.slice() : [];
  let photoIndex = 0;
  let slideshowTimerId = null;
  let pollTimerId = null;

  const formatMatch = (match) => {
    if (!match) return "Waiting for teams";
    if (Array.isArray(match.pool_group) && match.pool_group.length) {
      return match.pool_group.join(" + ");
    }
    if (match.pool_pair && match.team2) return `${match.team1} + ${match.team2}`;
    if (match.is_bye) return `${match.team1} (solo run)`;
    if (match.team2) {
      return `${match.team1} vs ${match.team2}`;
    }
    return match.team1 || "TBD";
  };

  const ensurePhotoElement = () => {
    if (photoEl) {
      return photoEl;
    }
    if (!frameEl) {
      return null;
    }
    const img = document.createElement("img");
    img.setAttribute("data-cast-photo", "true");
    img.loading = "lazy";
    img.alt = "Freeze Fest highlight";
    img.classList.add("cast-photo__image");
    frameEl.innerHTML = "";
    frameEl.appendChild(img);
    photoEl = img;
    placeholderEl = null;
    return photoEl;
  };

  const updateCaption = (photo) => {
    if (!captionEl) return;
    if (!photo) {
      captionEl.textContent = "Gallery will update automatically when new photos arrive.";
      return;
    }
    const label = photo.original_name || "Freeze Fest highlight";
    captionEl.textContent = label;
  };

  const showPhoto = (index) => {
    if (!photos.length) {
      if (placeholderEl) {
        placeholderEl.style.display = "flex";
      }
      if (photoEl) {
        photoEl.classList.add("is-hidden");
      }
      updateCaption(null);
      return;
    }

    const photo = photos[index];
    if (!photo) {
      return;
    }

    const target = ensurePhotoElement();
    if (!target) {
      return;
    }

    target.classList.remove("is-hidden");
    target.src = photo.image_url;
    target.alt = photo.original_name ? `${photo.original_name} – Freeze Fest highlight` : "Freeze Fest highlight";
    target.dataset.photoId = String(photo.id);
    updateCaption(photo);
  };

  const stopSlideshow = () => {
    if (slideshowTimerId) {
      clearInterval(slideshowTimerId);
      slideshowTimerId = null;
    }
  };

  const startSlideshow = () => {
    stopSlideshow();
    if (photos.length <= 1) {
      return;
    }
    slideshowTimerId = window.setInterval(() => {
      photoIndex = (photoIndex + 1) % photos.length;
      showPhoto(photoIndex);
    }, PHOTO_INTERVAL_MS);
  };

  const syncPhotos = (incoming) => {
    if (!Array.isArray(incoming)) {
      return;
    }

    const currentId = photos[photoIndex]?.id;
    photos = incoming.slice();

    if (!photos.length) {
      photoIndex = 0;
      showPhoto(photoIndex);
      stopSlideshow();
      return;
    }

    const existingIndex = photos.findIndex((photo) => photo.id === currentId);
    photoIndex = existingIndex >= 0 ? existingIndex : photos.length - 1;
    showPhoto(photoIndex);
    startSlideshow();
  };

  const syncLayoutHeights = () => {
    if (!layoutEl || !frameEl) {
      return;
    }
    const layoutStyles = window.getComputedStyle(layoutEl);
    const paddingTop = parseFloat(layoutStyles.paddingTop) || 0;
    const paddingBottom = parseFloat(layoutStyles.paddingBottom) || 0;
    const availableHeight = Math.max(360, window.innerHeight - paddingTop - paddingBottom);

    const gridHeight = gamesContainer?.scrollHeight || 0;
    const photoMaxHeight = gridHeight ? Math.min(gridHeight, availableHeight) : availableHeight;
    frameEl.style.maxHeight = `${photoMaxHeight}px`;
    frameEl.style.height = `${photoMaxHeight}px`;
  };

  const renderGames = (games) => {
    if (!gamesContainer) {
      return;
    }

    const clearGameCards = () => {
      const existing = gamesContainer.querySelectorAll("[data-game-card]");
      existing.forEach((node) => node.remove());
    };

    clearGameCards();

    if (!Array.isArray(games) || games.length === 0) {
      const card = document.createElement("article");
      card.className = "cast-card cast-card--empty";
      card.dataset.gameCard = "true";
      const title = document.createElement("h2");
      title.textContent = "Matches";
      const message = document.createElement("p");
      message.className = "cast-card__value";
      message.textContent = "Match assignments will appear here as soon as the tournament begins.";
      card.append(title, message);
      gamesContainer.appendChild(card);
      return;
    }

    games.forEach((game) => {
      const card = document.createElement("article");
      card.className = "cast-card";
      card.dataset.gameCard = "true";

      const heading = document.createElement("h2");
      heading.textContent = game.game;
      card.appendChild(heading);

      const playingSection = document.createElement("div");
      playingSection.className = "cast-card__section";
      const playingTitle = document.createElement("h3");
      playingTitle.textContent = "Playing Now";
      playingSection.appendChild(playingTitle);

      const playingSlots = Array.isArray(game.current) ? game.current : game.current ? [game.current] : [];
      if (playingSlots.length) {
        const list = document.createElement("ul");
        list.className = "cast-card__matches";
        playingSlots.forEach((slot) => {
          const item = document.createElement("li");
          item.textContent = formatMatch(slot);
          list.appendChild(item);
        });
        playingSection.appendChild(list);
      } else {
        const playingValue = document.createElement("p");
        playingValue.className = "cast-card__value";
        playingValue.textContent =
          game.remaining === 0 ? "All matches complete" : "Waiting for the next teams";
        playingSection.appendChild(playingValue);
      }
      card.appendChild(playingSection);

      const nextSection = document.createElement("div");
      nextSection.className = "cast-card__section";
      const nextTitle = document.createElement("h3");
      nextTitle.textContent = "Upcoming Games";
      nextSection.appendChild(nextTitle);

      const upcomingMatches = [];
      if (game.next) {
        upcomingMatches.push(game.next);
      }
      if (Array.isArray(game.upcoming_queue) && game.upcoming_queue.length) {
        upcomingMatches.push(...game.upcoming_queue);
      }

      if (upcomingMatches.length) {
        const queueList = document.createElement("ul");
        queueList.className = "cast-card__queue";
        upcomingMatches.forEach((match) => {
          const item = document.createElement("li");
          item.textContent = formatMatch(match);
          queueList.appendChild(item);
        });
        nextSection.appendChild(queueList);
      } else {
        const nextValue = document.createElement("p");
        nextValue.className = "cast-card__value";
        nextValue.textContent =
          game.remaining === 0 ? "Tournament finished for this game" : "Stand by for assignments";
        nextSection.appendChild(nextValue);
      }
      card.appendChild(nextSection);

      gamesContainer.appendChild(card);
    });

    syncLayoutHeights();
  };

  const renderLeaderboard = (teams, bucketMode) => {
    if (!leaderboardGrid || !leaderboardSection) {
      return;
    }
    if (!Array.isArray(teams) || !teams.length) {
      leaderboardSection.classList.add("is-hidden");
      leaderboardGrid.innerHTML = "";
      return;
    }
    leaderboardSection.classList.remove("is-hidden");
    leaderboardGrid.innerHTML = "";
    const chunkSize = 4;
    for (let start = 0; start < teams.length; start += chunkSize) {
      const column = document.createElement("div");
      column.className = "cast-semis__col";
      for (let offset = 0; offset < chunkSize; offset += 1) {
        const index = start + offset;
        const team = teams[index];
        if (!team) {
          continue;
        }
        const item = document.createElement("div");
        item.className = "cast-semis__item";
        if (index < 4) {
          item.classList.add("cast-semis__item--qualified");
        }

        const rank = document.createElement("span");
        rank.className = "cast-semis__rank";
        rank.textContent = `#${index + 1}`;
        item.appendChild(rank);

        const body = document.createElement("div");
        body.className = "cast-semis__body";
        const name = document.createElement("p");
        name.className = "cast-semis__name-title";
        name.textContent = team.name || "Team";
        body.appendChild(name);

        if (bucketMode) {
          const bucket = document.createElement("p");
          bucket.className = "cast-semis__meta cast-semis__golf";
          bucket.textContent =
            typeof team.bucket_score === "number" ? `Strokes: ${team.bucket_score}` : "Golf incomplete";
          body.appendChild(bucket);
        }

        const record = document.createElement("div");
        record.className = "cast-semis__record";
        if (typeof team.wins === "number" && typeof team.losses === "number") {
          const wl = document.createElement("p");
          wl.className = "cast-semis__meta";
          wl.textContent = `W: ${team.wins} / L: ${team.losses}`;
          record.appendChild(wl);
          if (team.ties) {
            const ties = document.createElement("p");
            ties.className = "cast-semis__meta";
            ties.textContent = `T: ${team.ties}`;
            record.appendChild(ties);
          }
        } else {
          const placeholder = document.createElement("p");
          placeholder.className = "cast-semis__meta";
          placeholder.textContent = "W/L/T: --";
          record.appendChild(placeholder);
        }

        body.appendChild(record);
        item.appendChild(body);
        column.appendChild(item);
      }
      if (column.children.length) {
        leaderboardGrid.appendChild(column);
      }
    }
  };

  const updateTimestamp = (isoString) => {
    if (!updatedEl) return;
    if (!isoString) {
      updatedEl.textContent = "Updated —";
      return;
    }
    const parsed = new Date(isoString);
    if (Number.isNaN(parsed.getTime())) {
      updatedEl.textContent = `Updated ${isoString}`;
      return;
    }
    const formatted = parsed.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    updatedEl.textContent = `Updated ${formatted}`;
    updatedEl.dataset.raw = isoString;
  };

  const applyState = (nextState) => {
    if (!nextState) {
      return;
    }
    syncPhotos(nextState.photos || []);
    renderGames(nextState.games || []);
    renderLeaderboard(nextState.semifinalists || [], nextState.bucket_pool_mode);
    updateTimestamp(nextState.generated_at);
    syncLayoutHeights();
  };

  const schedulePoll = () => {
    if (pollTimerId) {
      clearTimeout(pollTimerId);
    }
    pollTimerId = window.setTimeout(poll, POLL_INTERVAL_MS);
  };

  const poll = async () => {
    try {
      const response = await fetch(`/cast/feed?ts=${Date.now()}`, {
        headers: { Accept: "application/json" },
        credentials: "same-origin",
        cache: "no-store",
      });
      if (!response.ok) {
        throw new Error(`Cast feed request failed with status ${response.status}`);
      }
      const data = await response.json();
      applyState(data);
    } catch (error) {
      console.warn("Unable to refresh cast feed:", error);
    } finally {
      schedulePoll();
    }
  };

  // Initial render
  applyState(initialState);
  schedulePoll();
  window.addEventListener("resize", syncLayoutHeights);
  syncLayoutHeights();
})();
